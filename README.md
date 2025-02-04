# simple-chatbot-with-open-source-LLMs-using-Python-and-Hugging-Face
Intro: How does a chatbot work?
A chatbot is a computer program that takes a text input, and returns a corresponding text output.

Chatbots use a special kind of computer program called a transformer, which is like its brain. Inside this brain, there is something called a language model (LLM), which helps the chatbot understand and generate human-like responses. It looks at lots of examples of human conversations it has seen before to help it respond in a way that makes sense.

Transformers and LLMs work together within a chatbot to enable conversation. Here’s a simplified explanation of how they interact:

Input Processing: When you send a message to the chatbot, the transformer helps process your input. It breaks down your message into smaller parts and represents them in a way that the chatbot can understand. Each part is called a token.

Understanding Context: The transformer passes these tokens to the LLM, which is a language model trained on lots of text data. The LLM has learned patterns and meanings from this data, so it tries to understand the context of your message based on what it has learned.

Generating Response: Once the LLM understands your message, it generates a response based on its understanding. The transformer then takes this response and converts it into a format that can be easily sent back to you.

Iterative Conversation: As the conversation continues, this process repeats. The transformer and LLM work together to process each new input message, understand the context, and generate a relevant response.

The key is that the LLM learns from a large amount of text data to understand language patterns and generate meaningful responses. The transformer helps with the technical aspects of processing and representing the input/output data, allowing the LLM to focus on understanding and generating language

Once the chatbot understands your message, it uses the language model to generate a response that it thinks will be helpful or interesting to you. The response is sent back to you, and the process continues as you have a back-and-forth conversation with the chatbot.

Intro: Hugging Face
Hugging Face is an organization that focuses on natural language processing (NLP) and AI. They provide a variety of tools, resources, and services to support NLP tasks.

Step 1: Installing Requirements

!pip install transformers
Step 2: Import our required tools from the transformers library

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
Step 3: Choosing a model

Choosing the right model for your purposes is an important part of building chatbots! You can read on the different types of models available on the Hugging Face website: https://huggingface.co/models.

LLMs differ from each other in how they are trained. Let’s gloss over some examples to see how different models fit better in various contexts.

Text Generation: If you need a general-purpose text generation model, consider using the GPT-2 or GPT-3 models. They are known for their impressive language generation capabilities. Example: You want to build a chatbot that generates creative and coherent responses to user input.
Sentiment Analysis: For sentiment analysis tasks, models like BERT or RoBERTa are popular choices. They are trained to understand the sentiment and emotional tone of text. Example: You want to analyze customer feedback and determine whether it is positive or negative.
Named Entity Recognition: LLMs such as BERT, GPT-2, or RoBERTa can be used for Named Entity Recognition (NER) tasks. They perform well in understanding and extracting entities like person names, locations, organizations, etc. Example: You want to build a system that extracts names of people and places from a given text.
Question Answering: Models like BERT, GPT-2, or XLNet can be effective for question answering tasks. They can comprehend questions and provide accurate answers based on the given context. Example: You want to build a chatbot that can answer factual questions from a given set of documents.
Language Translation: For language translation tasks, you can consider models like MarianMT or T5. They are designed specifically for translating text between different languages. Example: You want to build a language translation tool that translates English text to French.
However, these examples are very limited and the fit of an LLM may depend on many factors such as data availability, performance requirements, resource constraints, and domain-specific considerations. It’s important to explore different LLMs thoroughly and experiment with them to find the best match for your specific application.

Other important purposes that should be taken into consideration when choosing an LLM include (but are not limited to):

Licensing: Ensure you are allowed to use your chosen model the way you intend
Model size: Larger models may be more accurate, but might also come at the cost of greater resource requirements
Training data: Ensure that the model’s training data aligns with the domain or context you intend to use the LLM for
Performance and accuracy: Consider factors like accuracy, runtime, or any other metrics that are important for your specific use case.
model_name = "facebook/blenderbot-400M-distill"
Step 4: Fetch the model and initialize a tokenizer

When running this code for the first time, the host machine will download the model from Hugging Face API. However, after running the code once, the script will not re-download the model and will instead reference the local installation.

WE are looking for two terms :model and tokenizer.

In this script , variables using two handy classes from the transformers library are initiated:

model is an instance of the class ,AutoModelForSeq2SeqLM,whixh allows us to interact with our chosen language models.
tokenizer is an instance of the class AutoTokenizer,which optimizes our input and passes it to the language model efficiently .It does so by converting out text input to tokens which is how the model interperts the text.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
Step 5: Chat

Now that we’re all set up, let’s start chatting!There are several things we’ll do to have an effective conversation with our chatbot.Before interacting with our model, we need to initialize an object where we can store our conversation history.

Initialize object to store conversation history
Afterwards, we’ll do the following for each interaction with the model:

2. Encode conversation history as a string

3. Fetch prompt from user

4. Tokenize (optimize) prompt

5. Generate output from model using prompt and history

6. Decode output

7. Update conversation history

Step 5.1: Keeping track of conversation history

The conversation history is important when interacting with a chatbot because the chatbot will also reference the previous conversations when generating output.

For our simple implementation in Python, we may simply use a list. Per the Hugging Face implementation, we will use this list to store the conversation history as follows:

conversation_history

>> [input_1, output_1, input_2, output_2, ...]
conversation_history = []
Step 5.2: Encoding the conversation history

During each interaction, we will pass our conversation history to the model along with our input so that it may also reference the previous conversation when generating the next answer.

The transformers library function we are using expects to receive the conversation history as a string, with each element separated by the newline character ‘\n’ . Thus, we create such a string.

We’ll use the join() method in Python to do exactly that. (Initially, our history_string will be an empty string, which is okay, and will grow as the conversation goes on)

history_string = "\n".join(conversation_history)
history_string 
Step 5.3: Fetch prompt from user

Befor we start building a simple terminal chatbot, let’s example, the input will be

input_text ="hello"
input_text
Step 5.4: Tokenization of User Prompt and Chat History

Tokens in NLP are individual units or elements that text or sentences are divided into. Tokenization or vectorization is the process of converting tokens into numerical representations. In NLP tasks, we often use the encode_plus method from the tokenizer object to perform tokenization and vectorization. Let's encode our inputs (prompt & chat history) as tokens so that we may pass them to the model.

inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
inputs
In doing so, we’ve now created a Python dictinoary which contains special keywords that allow the model to properly reference its contents. To learn more about tokens and their associated pretrained vocabulary files, you can explore the pretrained_vocab_files_map attribute. This attribute provides a mapping of pretrained models to their corresponding vocabulary files.

tokenizer.pretrained_vocab_files_map
Step 5.5: Generate output from model

Now that we have our inputs ready, both past and present inputs, we can pass them to the model and generate a response. According to the documentation, we can use the generate() function and pass the inputs as keyword arguments (kwargs).

output = model.generate(**inputs)
output
Great — now we have our outputs! However, the current output is also a dictionary and contains tokens, not words in plaintext. Therefore, we just need to decode the first index of output to see the response in plaintext.

Step 5.6: Decode output

We may decode the output using tokenizer.decode() . This is know as "detokenization" or "reconstruction". It is the process of combining or merging individual tokens back into their original form, typically to reconstruct the original text or sentence

response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
response
hell hell hell Hello Hello Hello hell hell hi hi hi Hi hi hi hell hell hey hell hell Hell hell hell'
Alright! We’ve successfully had an interaction with our chatbot! We’ve given it a prompt, and we received its response.

Step 5.7: Update Conversation History

All we need to do here is add both the input and response to conversation_history in plaintext.

conversation_history.append(input_text)
conversation_history.append(response)
conversation_history
