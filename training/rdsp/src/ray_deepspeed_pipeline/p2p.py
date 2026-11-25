# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Direct stage-to-stage tensor exchange for stage-local dispatch.

Each rank joins two process groups spanning every rank of every stage: one
carries activations downstream, the other gradients upstream. They are
standalone groups built from a TCPStore, so the default torch.distributed
world stays the stage-local world DeepSpeed owns.

Why two groups: NCCL runs a communicator's point-to-point ops in issue order,
so with one shared group a stage's gradient receive could queue behind its
own activation send (and its neighbour's behind theirs) and deadlock. One
direction per group keeps each group's dependencies acyclic. Sends are async
and kept alive until the step ends; receives block, and 1F1B order
guarantees each one's matching send is eventually issued.

When a step's microbatches are equal-sized, each (direction, peer, slot)
sends its shape header once per step; both ends walk microbatches in the same
order, so they agree on which message carries it. In a step whose
microbatches differ in length (each padded to its own longest row),
begin_step(varying=True) makes every message carry its header. A forward
boundary is the hidden state (slot 0) plus named extra tensors (slots 2+); the
hidden state's header carries their count and slot 1 their names, once per
step.
"""

import datetime

import torch
import torch.distributed as dist

from ray_deepspeed_pipeline.errors import ValidationError

_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.int64,
    torch.bool,
    torch.int32,
    torch.int16,
]
_HEADER_LEN = 10  # dtype code, ndim, up to 7 dims, extras count
_SLOTS = 64  # messages per microbatch and direction: hidden, extras' names, extras
_SIZED_TAGS = 1 << 24  # CPU-group messages (colocated vision) use tags from here on


def _tag(mb: int, slot: int) -> int:
    return 2 * (mb * _SLOTS + slot)


def _encode(t: torch.Tensor, n_extras: int = 0) -> torch.Tensor:
    header = [_DTYPES.index(t.dtype), t.dim(), *t.shape]
    if len(header) >= _HEADER_LEN:
        raise ValidationError(f"boundary tensors have at most 7 dims, got {t.dim()}")
    header += [0] * (_HEADER_LEN - 1 - len(header)) + [n_extras]
    return torch.tensor(header, dtype=torch.int64)


def _decode(header: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], int]:
    vals = header.tolist()
    return _DTYPES[vals[0]], tuple(vals[2 : 2 + vals[1]]), vals[-1]


class PipelineP2P:
    """One rank's two cross-stage groups, plus per-step bookkeeping.
    collective: also, for colocated vision, `self.all` (its gradient sum) and
    `self.cpu`, a gloo group for its features and feature gradients: a
    message there can wait any time for its receiver without holding up the
    GPU, which an NCCL send does."""

    def __init__(
        self,
        store,
        rank: int,
        world: int,
        epoch: int,
        timeout_s: float,
        device: torch.device,
        collective: bool = False,
    ):
        self.rank, self.world, self.device = rank, world, device
        timeout = datetime.timedelta(seconds=timeout_s)
        self.fwd = self._group(dist.PrefixStore(f"e{epoch}/fwd", store), timeout)
        self.bwd = self._group(dist.PrefixStore(f"e{epoch}/bwd", store), timeout)
        self.all = self.cpu = None
        if collective:
            self.all = self._group(dist.PrefixStore(f"e{epoch}/all", store), timeout)
            self.cpu = dist.ProcessGroupGloo(
                dist.PrefixStore(f"e{epoch}/cpu", store), rank, world, timeout
            )
        self._pending = []  # (work, tensor) sends in flight this step
        self._shapes = {}  # (direction, peer, slot) -> decoded header, this step
        self._varying = False  # this step's microbatches differ in length
        self._announced = set()  # (direction, peer, slot) already sent a header this step
        self._names = {}  # (direction, peer) -> extras' names, this step

    def _group(self, store, timeout):
        if self.device.type == "cuda":
            opts = dist.ProcessGroupNCCL.Options()
            opts._timeout = timeout
            return dist.ProcessGroupNCCL(store, self.rank, self.world, opts)
        return dist.ProcessGroupGloo(store, self.rank, self.world, timeout)

    def abort(self) -> None:
        """Tear down both groups now. After a failed step an unmatched send can
        sit on an NCCL stream; aborting releases it before the watchdog would
        kill the process."""
        self._pending.clear()
        for pg in (self.fwd, self.bwd, self.all, self.cpu):
            if pg is None:
                continue
            for name in ("abort", "shutdown"):
                fn = getattr(pg, name, None)
                if fn is not None:
                    try:
                        fn()
                        break
                    except Exception:
                        pass

    # -- per step --------------------------------------------------------------

    def begin_step(self, varying: bool = False) -> None:
        self._varying = varying
        self._shapes.clear()
        self._announced.clear()
        self._names.clear()

    def end_step(self) -> None:
        """Wait for every send of this step, so nothing crosses into the next."""
        pending, self._pending = self._pending, []
        for work, _ in pending:
            work.wait()

    # -- transfers -------------------------------------------------------------

    def _pg(self, direction: str):
        return self.fwd if direction == "fwd" else self.bwd

    def send(
        self, direction: str, t: torch.Tensor, peer: int, mb: int, slot: int = 0, n_extras: int = 0
    ) -> None:
        pg = self._pg(direction)
        t = t.detach().contiguous()
        if (direction, peer, slot) not in self._announced or (self._varying and slot != 1):
            self._announced.add((direction, peer, slot))
            header = _encode(t, n_extras).to(self.device)
            self._pending.append((pg.send([header], peer, _tag(mb, slot)), header))
        self._pending.append((pg.send([t], peer, _tag(mb, slot) + 1), t))

    def recv(self, direction: str, peer: int, mb: int, slot: int = 0) -> torch.Tensor:
        pg = self._pg(direction)
        key = (direction, peer, slot)
        if key not in self._shapes or (self._varying and slot != 1):
            header = torch.empty(_HEADER_LEN, dtype=torch.int64, device=self.device)
            pg.recv([header], peer, _tag(mb, slot)).wait()
            self._shapes[key] = _decode(header.cpu())
        dtype, shape, _ = self._shapes[key]
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        pg.recv([buf], peer, _tag(mb, slot) + 1).wait()
        return buf

    def send_cpu(self, t: torch.Tensor, peer: int, key: int) -> None:
        """A tensor over the gloo group, staged through host memory, with its
        header; returns at once. key: a per-step message number both ends
        agree on."""
        t = t.detach().to("cpu").contiguous()
        bf16 = t.dtype == torch.bfloat16  # sent as its bits: gloo may lack bf16
        if bf16:
            t = t.view(torch.int16)
        header, tag = _encode(t, n_extras=int(bf16)), _SIZED_TAGS + 2 * key
        self._pending.append((self.cpu.send([header], peer, tag), header))
        self._pending.append((self.cpu.send([t], peer, tag + 1), t))

    def recv_cpu(self, peer: int, key: int) -> torch.Tensor:
        """The matching receive, onto this rank's device."""
        tag = _SIZED_TAGS + 2 * key
        header = torch.empty(_HEADER_LEN, dtype=torch.int64)
        self.cpu.recv([header], peer, tag).wait()
        dtype, shape, bf16 = _decode(header)
        buf = torch.empty(shape, dtype=dtype)
        self.cpu.recv([buf], peer, tag + 1).wait()
        if bf16:
            buf = buf.view(torch.bfloat16)
        return buf.to(self.device)

    def send_boundary(
        self, direction: str, hidden: torch.Tensor, extras: dict, peer: int, mb: int
    ) -> None:
        """The hidden state plus named extra tensors, which carry no gradient."""
        if len(extras) > _SLOTS - 2:
            raise ValidationError(f"at most {_SLOTS - 2} extra tensors, got {len(extras)}")
        self.send(direction, hidden, peer, mb, n_extras=len(extras))
        if extras and (direction, peer, 1) not in self._announced:
            names = list(",".join(extras).encode())
            self.send(
                direction,
                torch.tensor(names, dtype=torch.int64, device=self.device),
                peer,
                mb,
                slot=1,
            )
        for i, t in enumerate(extras.values()):
            self.send(direction, t, peer, mb, slot=2 + i)

    def recv_boundary(self, direction: str, peer: int, mb: int) -> tuple[torch.Tensor, dict]:
        hidden = self.recv(direction, peer, mb)
        if not self._shapes[(direction, peer, 0)][2]:
            return hidden, {}
        if (direction, peer) not in self._names:
            names = self.recv(direction, peer, mb, slot=1).tolist()
            self._names[(direction, peer)] = bytes(names).decode().split(",")
        return hidden, {
            name: self.recv(direction, peer, mb, slot=2 + i)
            for i, name in enumerate(self._names[(direction, peer)])
        }


def make_store(host: str, port: int, world: int, is_master: bool, timeout_s: float):
    return dist.TCPStore(
        host,
        port,
        world,
        is_master,
        timeout=datetime.timedelta(seconds=timeout_s),
        wait_for_workers=False,
    )
