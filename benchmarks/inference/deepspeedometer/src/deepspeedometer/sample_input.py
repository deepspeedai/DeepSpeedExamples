# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This is a sample input consisting of:
# Code & Text

sample_input_text = """Deep learning involves the use of neural networks, which are computational models inspired by the structure and functioning of the human brain. These networks consist of interconnected nodes called neurons. Each neuron takes input, performs a computation, and produces an output.
              During training, the neural network learns to make accurate predictions by adjusting its internal parameters. This adjustment is done using an optimization algorithm called gradient descent. Gradient descent calculates the gradients of a loss function, which measures the discrepancy between the predicted output of the network and the desired output. These gradients indicate the direction and magnitude of parameter updates that will minimize the loss.
              The learning rate is an important hyperparameter in gradient descent. It determines the step size taken during parameter updates. A higher learning rate can lead to faster convergence, but it risks overshooting the optimal solution. On the other hand, a lower learning rate may converge more slowly, but it can result in more precise updates.
              Activation functions are applied to the output of each neuron in a neural network. They introduce non-linearities, enabling the network to learn complex patterns and relationships in the data. Popular activation functions include the rectified linear unit (ReLU), sigmoid, and hyperbolic tangent (tanh).
              By adjusting the parameters of the neural network during training, deep learning models learn to represent and generalize from complex data patterns. They have achieved remarkable success in various tasks, including image recognition, speech recognition, and natural language processing.
              Here are the key fundamentals of deep learning for training large language models:
              Neural Networks: At the heart of deep learning are artificial neural networks, which are inspired by the structure and functioning of biological neurons in the human brain. These networks consist of interconnected layers of artificial neurons called nodes or units. The nodes receive input, perform computations, and pass the results to the next layer.
              Representation Learning: Deep learning models excel at learning meaningful representations of data. In the context of language, the models can automatically learn hierarchical representations of text, capturing complex relationships and semantic structures.
              Feedforward and Backpropagation: Deep learning models typically use feedforward neural networks, where information flows from the input layer through intermediate hidden layers to the output layer. The network makes predictions based on the input data, and the prediction error is then backpropagated through the network. Backpropagation calculates gradients that indicate how each parameter in the network should be adjusted to minimize the error.
              Activation Functions: Activation functions introduce non-linearities to neural networks, enabling them to learn complex patterns. Common activation functions include the rectified linear unit (ReLU), sigmoid, and hyperbolic tangent (tanh). These functions determine the output of each neuron based on its weighted inputs.
              Loss Functions: During training, a loss function is used to measure the discrepancy between the predicted output of the neural network and the desired output. In language modeling tasks, common loss functions include cross-entropy loss, which quantifies the difference in probability distributions.
              Optimization Algorithms: Optimization algorithms determine how the network's parameters are updated based on the calculated gradients during backpropagation. Stochastic Gradient Descent (SGD) is a widely used algorithm that iteratively updates the parameters in the direction that minimizes the loss. Variants of SGD, such as Adam or RMSprop, adaptively adjust the learning rate to accelerate convergence.
              Regularization Techniques: Deep learning models are prone to overfitting, where they memorize the training data but fail to generalize well to unseen examples. Regularization techniques such as dropout and weight decay are commonly used to prevent overfitting and improve generalization by adding constraints to the model's parameters.
              Training on Large-Scale Datasets: Deep learning models, including large language models, require substantial amounts of labeled training data to learn effectively. Large-scale datasets are crucial to expose the model to diverse language patterns and ensure it captures a broad understanding of language.
              Parallel Computing: Training large language models is computationally demanding. To accelerate the training process, parallel computing techniques, such as using multiple GPUs or distributed computing systems, are employed. These techniques allow for efficient processing of large datasets and speeding up the training iterations.
              Transfer Learning and Fine-tuning: Transfer learning is a technique where a pre-trained model, trained on a large-scale dataset, is used as a starting point for a new task or dataset. Fine-tuning involves adjusting the pre-trained model's parameters on the new dataset to adapt it to the specific task at hand. This approach significantly reduces the training time and data requirements for new models.
              The training process of a large language model typically involves the following steps:
              Data Collection: A diverse and comprehensive dataset is collected, which typically consists of a vast range of text from sources like books, websites, articles, and other textual resources. The quality and variety of the dataset are crucial to ensure the model learns a broad understanding of language.
              Preprocessing: The collected text data is preprocessed to clean and normalize it. This step involves removing irrelevant characters or symbols, converting the text to a consistent format, and organizing it into smaller units such as sentences or paragraphs.
              Tokenization: The preprocessed text is divided into individual tokens, which can be as small as words or even subword units. Tokenization helps in representing and processing the text efficiently during training.
              Architecture Design: The model architecture, often based on the transformer architecture, is defined. Transformers are neural network models that excel in capturing long-range dependencies in sequential data, making them well-suited for language modeling tasks.
              Model Initialization: The model parameters are randomly initialized to start the training process. These parameters will be adjusted iteratively during training to optimize the model's performance.
              Training Loop: The model is trained using a large-scale computational infrastructure. The training loop typically involves several iterations over the dataset, known as epochs. During each epoch, the model processes the input data, generates predictions, and compares them with the expected output. The discrepancy between the predicted and expected output is used to compute a loss, which quantifies the model's performance.
              Backpropagation and Optimization: Backpropagation is employed to calculate the gradients of the model's parameters with respect to the loss. These gradients indicate the direction and magnitude of the parameter updates needed to minimize the loss. Optimization algorithms, such as stochastic gradient descent (SGD) or its variants, are then used to update the model's parameters based on the computed gradients.
              Iterative Refinement: Steps 6 and 7 are repeated for multiple epochs, gradually refining the model's performance. The model's ability to generate coherent and contextually relevant responses improves as it learns from the dataset.
              Evaluation: The trained model is evaluated on a separate dataset to assess its performance and identify areas for improvement. Various metrics, such as perplexity or accuracy, can be used to evaluate the model's language generation capabilities.
              Fine-tuning and Iteration: Based on the evaluation results, the model may undergo fine-tuning or further iterations of training to enhance its performance. This process helps in addressing specific limitations or biases and aligning the model's output more closely with desired expectations.
              It's important to note that training a large language model from scratch is a computationally intensive process that requires substantial computational resources, including powerful hardware like GPUs or specialized hardware accelerators, and large-scale distributed systems to handle the massive amount of data and model parameters involved.
              Here are ten highly recommended books that can help you learn deep learning:
              "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:
              This comprehensive book covers the fundamental concepts of deep learning, including neural networks, optimization algorithms, and regularization techniques. It also explores advanced topics like generative models and deep reinforcement learning.
              "Deep Learning with Python" by François Chollet:
              Written by the creator of the Keras deep learning library, this book provides a practical introduction to deep learning with Python. It covers essential concepts, tools, and techniques, and includes hands-on examples and case studies.
              "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron:
              This book offers a hands-on approach to learning machine learning and deep learning using popular Python libraries such as Scikit-Learn, Keras, and TensorFlow. It covers various algorithms and provides practical examples and exercises.
              "Deep Learning for Computer Vision" by Rajalingappaa Shanmugamani:
              Focusing on deep learning techniques for computer vision tasks, this book explores topics such as convolutional neural networks (CNNs), image classification, object detection, and image generation. It includes code examples using Python and popular deep learning frameworks.
              "Deep Learning: A Practitioner's Approach" by Josh Patterson and Adam Gibson:
              This book offers a practical guide to implementing deep learning solutions using the Deeplearning4j library. It covers key concepts, architectures, and techniques, and includes code examples and case studies.
              "Grokking Deep Learning" by Andrew Trask:
              Geared towards beginners, this book provides an intuitive and accessible introduction to deep learning concepts. It covers neural networks, backpropagation, gradient descent, and other fundamental topics with clear explanations and visualizations.
              "Deep Learning for Natural Language Processing" by Palash Goyal, Sumit Pandey, and Karan Jain:
              Focusing on deep learning techniques for natural language processing (NLP), this book explores topics like word embeddings, recurrent neural networks (RNNs), and sequence-to-sequence models. It includes code examples using Python and popular NLP libraries.
              "Deep Reinforcement Learning" by Pieter Abbeel and John Schulman:
              This book provides an in-depth exploration of deep reinforcement learning, a subfield that combines deep learning with reinforcement learning. It covers topics like Q-learning, policy gradients, and deep Q-networks (DQNs) and provides practical examples.
              "Deep Learning for Time Series Forecasting" by N.D. Lewis:
              Focusing on deep learning techniques for time series data, this book covers topics such as recurrent neural networks (RNNs), long short-term memory (LSTM) networks, and attention models. It includes code examples using Python and popular deep learning frameworks.
              "Interpretable Deep Learning" by Christoph Molnar:
              This book delves into the challenges and techniques for interpreting and understanding deep learning models. It covers model visualization, feature importance, and other methods for explaining and interpreting deep learning predictions.
              These books cover a range of deep learning topics and provide valuable insights and practical guidance for learning and applying deep learning techniques. Choose the ones that align with your interests and learning style to enhance your understanding of deep learning.
              Here are 10 popular GitHub projects that can be useful for building large language models (LLMs) or working with natural language processing (NLP) tasks:
              TensorFlow: An open-source deep learning framework that provides tools and resources for building and training LLMs. It offers extensive support for various neural network architectures and has a large community.
              PyTorch: Another popular deep learning framework that provides a dynamic computational graph and a wide range of tools for building LLMs. It is known for its user-friendly interface and flexibility.
              Hugging Face Transformers: A library that provides pre-trained models and a high-level API for natural language understanding (NLU) tasks, including LLMs. It supports popular models like GPT, BERT, and RoBERTa.
              Fairseq: A library developed by Facebook AI Research that focuses on sequence modeling tasks, including LLMs. It offers pre-trained models and tools for training and evaluating models using sequence-to-sequence architectures.
              AllenNLP: A powerful NLP research library that simplifies the process of building and evaluating deep learning models. It offers pre-built components for common NLP tasks and supports LLMs with various architectures.
              OpenAI GPT-3: Although not available on GitHub, OpenAI's GPT-3 language model is widely recognized and can be accessed via the OpenAI API. It offers state-of-the-art language generation capabilities and can be used for various NLP tasks.
              BERT: A pre-trained language model developed by Google Research that has achieved exceptional results on various NLP benchmarks. The official implementation is available on GitHub and can be fine-tuned for specific tasks.
              spaCy: A popular Python library for NLP tasks that provides efficient and scalable tools for tokenization, named entity recognition, part-of-speech tagging, and more. It integrates well with deep learning frameworks.
              FastText: A library developed by Facebook Research that provides efficient tools for text classification and word representation learning. It offers pre-trained word embeddings and supports training LLMs for classification tasks.
              NLTK (Natural Language Toolkit): A comprehensive library for NLP tasks in Python. It provides various modules for tokenization, stemming, tagging, parsing, and more. Although it doesn't focus explicitly on LLMs, it is widely used for preprocessing text data in NLP pipelines.
              These projects offer a range of resources, pre-trained models, and tools that can assist you in building and working with large language models. Make sure to review the documentation and examples provided by each project to understand their capabilities and how they can be integrated into your workflow.
              Here are some popular backend libraries that are commonly used for deep learning:
              TensorFlow: Developed by Google's Brain Team, TensorFlow is one of the most widely used deep learning frameworks. It provides a flexible and comprehensive ecosystem for building and deploying machine learning models. TensorFlow offers high-level APIs for easy model construction, as well as lower-level APIs for fine-grained control. It supports distributed computing and has extensive community support.
              PyTorch: Developed by Facebook's AI Research lab, PyTorch is known for its simplicity and dynamic computational graph. It allows for intuitive model construction and debugging. PyTorch is widely used in both research and industry due to its flexibility, support for dynamic networks, and strong GPU acceleration capabilities.
              Keras: Initially developed as a user-friendly deep learning library, Keras is now integrated as the official high-level API in TensorFlow. It provides a user-friendly and modular interface for building neural networks. Keras abstracts away many complexities and allows users to build models with just a few lines of code. It supports multiple backends, including TensorFlow and Theano.
              Theano: Although its development has been discontinued, Theano was one of the first widely-used deep learning libraries. It allows for efficient mathematical operations on multi-dimensional arrays and supports GPU acceleration. Theano was influential in shaping the deep learning landscape and served as a precursor to subsequent frameworks.
              Caffe: Developed by the Berkeley Vision and Learning Center (BVLC), Caffe is a popular deep learning framework known for its efficiency and simplicity. It is particularly suitable for convolutional neural networks (CNNs) and image-related tasks. Caffe has a clean and expressive architecture description language that makes it easy to define and train deep models.
              MXNet: MXNet is an open-source deep learning framework developed by Apache. It offers a flexible and efficient interface for building and deploying neural networks. MXNet provides a hybrid frontend that allows users to seamlessly switch between symbolic and imperative programming. It is known for its scalability and supports multiple programming languages.
              Chainer: Chainer is a flexible deep learning framework that focuses on dynamic neural networks. It allows for intuitive model construction using imperative programming, making it easy to define complex architectures and manipulate data within the network. Chainer is known for its "define-by-run" approach, which facilitates dynamic computations.
              Microsoft Cognitive Toolkit (CNTK): CNTK is a deep learning framework developed by Microsoft. It provides a highly efficient and scalable implementation of deep neural networks. CNTK supports both declarative and imperative programming models, making it suitable for both research and production-level deployments.
              Deeplearning4j: Deeplearning4j is an open-source deep learning library that focuses on scalability and performance. It is designed to integrate with the Java ecosystem and supports distributed computing. Deeplearning4j provides tools for building various types of neural networks and offers integration with other popular libraries like Hadoop and Spark.
              PaddlePaddle: PaddlePaddle (PArallel Distributed Deep LEarning) is a deep learning framework developed by Baidu. It emphasizes scalability and supports large-scale distributed training. PaddlePaddle provides a rich set of built-in models and algorithms, making it accessible to both beginners and advanced users.
              Each of these backend libraries offers unique features, performance characteristics, and levels of abstraction. The choice of a backend library depends on factors such as your programming language preferences, the complexity of your models, the availability of community support, and the specific requirements of your deep learning project.
              Here's an example code snippet that demonstrates how to create a GPT-Neox20B model using the Hugging Face Transformers library and start fine-tuning it with sample data from the '/tmp/wikitext' directory:
  
                import torch
                from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    
                # Load the GPT-Neo model and tokenizer
                model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
                tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    
                # Set the path to the training data
                data_path = "/tmp/wikitext"
    
                # Define the dataset and data collator
                dataset = TextDataset(tokenizer=tokenizer, file_path=data_path, block_size=128)
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
                # Define the training arguments
                training_args = TrainingArguments(
                    output_dir="./output_dir",  # Directory to save the fine-tuned model and logs
                    overwrite_output_dir=True,
                    num_train_epochs=3,  # Number of training epochs
                    per_device_train_batch_size=8,
                    save_steps=500,  # Number of updates steps before saving the model
                    save_total_limit=2,  # Limit the total number of saved models
                    prediction_loss_only=True,
                )
    
                # Initialize the Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=dataset,
                )
    
                # Start fine-tuning
                trainer.train()
    
                # Save the fine-tuned model
                trainer.save_model("./fine-tuned-model")

              To deploy the GPT-Neo model on an HTTP server, you can use a framework like Flask or FastAPI. Here's a basic example using Flask:

                Install Flask:

                pip install flask

            Create a Python file, let's call it app.py, and import the necessary libraries:

                from flask import Flask, request, jsonify
                from transformers import GPTNeoForCausalLM, GPT2Tokenizer

                # Initialize Flask app
                app = Flask(__name__)

                # Load the pretrained GPT-Neo model and tokenizer
                model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
                tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
                Define an endpoint for model inference:

                python
                Copy code
                @app.route("/generate", methods=["POST"])
                def generate_text():
                    data = request.get_json()
                    input_text = data["input_text"]

                    # Tokenize the input text
                    input_ids = tokenizer.encode(input_text, return_tensors="pt")

                    # Generate text with the model
                    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                    return jsonify({"generated_text": generated_text})
                Run the Flask app:

                python
                Copy code
                if __name__ == "__main__":
                    app.run(host="0.0.0.0", port=5000)
            The code that includes the text generation sampling functions and provides a commented example:

                import requests
                import numpy as np

                class TextGeneratorAPI:
                    def __init__(self, server_url):
                        self.server_url = server_url

                    def generate_text(self, input_text, sampling_algorithm="greedy", temperature=0.7):
                        url = f"{self.server_url}/generate"
                        payload = {
                            "input_text": input_text,
                            "sampling_algorithm": sampling_algorithm,
                            "temperature": temperature
                        }
                        response = requests.post(url, json=payload)
                        generated_text = response.json()["generated_text"]
                        return generated_text

                    def greedy_sampling(self, logits):
                        return np.argmax(logits)

                    def random_sampling(self, logits):
                        probabilities = np.exp(logits / temperature)
                        probabilities = probabilities / np.sum(probabilities)
                        return np.random.choice(len(logits), p=probabilities)

                    def top_k_sampling(self, logits, k=10):
                        indices = np.argsort(logits)[-k:]
                        probabilities = np.exp(logits[indices] / temperature)
                        probabilities = probabilities / np.sum(probabilities)
                        return np.random.choice(indices, p=probabilities)

                    def top_p_sampling(self, logits, p=0.9):
                        sorted_logits = np.sort(logits)[::-1]
                        cumulative_probs = np.cumsum(np.exp(sorted_logits) / temperature)
                        indices = np.arange(len(sorted_logits))
                        selected_indices = indices[cumulative_probs <= p]
                        probabilities = np.exp(logits[selected_indices] / temperature)
                        probabilities = probabilities / np.sum(probabilities)
                        return np.random.choice(selected_indices, p=probabilities)
                In this updated code, the TextGeneratorAPI class includes the additional sampling functions: greedy_sampling, random_sampling, top_k_sampling, and top_p_sampling. These functions take logits (output of the model) as input and return the index of the selected token based on the respective sampling algorithm.
                The greedy_sampling function selects the token with the highest probability (argmax) as the next token. The random_sampling function applies a temperature scaling to the logits and then samples from the resulting probability distribution. The top_k_sampling function selects from the top-k tokens with the highest probabilities. The top_p_sampling function selects from the tokens with cumulative probabilities below a certain threshold (top-p).
                You can now use the updated TextGeneratorAPI class with the sampling functions. Here's an example:

                    api = TextGeneratorAPI(server_url="http://localhost:5000")

                    input_text = "Once upon a time"

                    # Generate text using different sampling algorithms and temperatures
                    greedy_text = api.generate_text(input_text, sampling_algorithm="greedy")
                    random_text = api.generate_text(input_text, sampling_algorithm="random")
                    top_k_text = api.generate_text(input_text, sampling_algorithm="top_k", temperature=0.8)
                    top_p_text = api.generate_text(input_text, sampling_algorithm="top_p", temperature=0.9)

                    print("Greedy Sampling:", greedy_text)
                    print("Random Sampling:", random_text)
                    print("Top-k Sampling:", top_k_text)
                    print("Top-p Sampling:", top_p_text)
                    Make sure to adjust the server_url with the appropriate URL of your HTTP server, and ensure that the server is running and accessible before making requests through the API.
           """
