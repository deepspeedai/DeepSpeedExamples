import pytest


@pytest.fixture(scope="function", params=["facebook/opt-125m"])
def model(request):
    return request.param


@pytest.fixture(scope="function", params=["dummy"])
def api(request):
    return request.param


@pytest.fixture(scope="function", params=[""])
def result_dir(request, tmpdir):
    if request.param:
        return str(request.param)
    return str(tmpdir)


@pytest.fixture(scope="function", params=[5])
def num_requests_per_client(request):
    return str(request.param)


@pytest.fixture(scope="function", params=[16])
def min_requests(request):
    return str(request.param)


@pytest.fixture(scope="function", params=[(1, 2)])
def num_clients(request):
    if isinstance(request.param, tuple) or isinstance(request.param, list):
        return [str(num) for num in request.param]
    else:
        return [str(request.param)]


@pytest.fixture(scope="function", params=[0])
def num_config_files(request):
    return request.param


@pytest.fixture(scope="function")
def config_files(num_config_files, tmp_path):
    config_files = []
    for i in range(num_config_files):
        config_file = tmp_path / f"config_{i}.yaml"
        config_file.touch()
        config_files.append(str(config_file))
    return config_files


@pytest.fixture(scope="function", params=[""])
def prompt_length_var(request):
    return str(request.param)


@pytest.fixture(scope="function", params=[""])
def max_new_tokens_var(request):
    return str(request.param)


@pytest.fixture(scope="function")
def benchmark_args(
    model,
    api,
    result_dir,
    num_requests_per_client,
    min_requests,
    num_clients,
    config_files,
    prompt_length_var,
    max_new_tokens_var,
):
    args = []
    if model:
        args.extend(["--model", model])
    if api:
        args.extend(["--api", api])
    if result_dir:
        args.extend(["--result_dir", result_dir])
    if num_requests_per_client:
        args.extend(["--num_requests_per_client", num_requests_per_client])
    if min_requests:
        args.extend(["--min_requests", min_requests])
    if num_clients:
        args.extend(["--num_clients"] + num_clients)
    if config_files:
        args.extend(["--config_file"] + config_files)
    if prompt_length_var:
        args.extend(["--prompt_length_var", prompt_length_var])
    if max_new_tokens_var:
        args.extend(["--max_new_tokens_var", max_new_tokens_var])
    return args
