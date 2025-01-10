import os

import boto3
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
import json
class AwsProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the AWS Bedrock provider with the given configuration.

        This class uses the AWS Bedrock converse API, which provides a consistent interface
        for all Amazon Bedrock models that support messages. Examples include:
        - anthropic.claude-v2
        - meta.llama3-70b-instruct-v1:0
        - mistral.mixtral-8x7b-instruct-v0:1

        The model value can be a baseModelId for on-demand throughput or a provisionedModelArn
        for higher throughput. To obtain a provisionedModelArn, use the CreateProvisionedModelThroughput API.

        For more information on model IDs, see:
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html

        Note:
        - The Anthropic Bedrock client uses default AWS credential providers, such as
          ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
        - If the region is not set, it defaults to us-west-1, which may lead to a
          "Could not connect to the endpoint URL" error.
        - The client constructor does not accept additional parameters.

        Args:
            **config: Configuration options for the provider.

        """
        self.region_name = config.get(
            "region_name", os.getenv("AWS_REGION", "us-west-2")
        )


        filtered_creds = {
            'aws_access_key_id': config['aws_access_key_id'],
            'aws_secret_access_key': config['aws_secret_access_key']
        }

        print(self.region_name)

        self.client = boto3.client("bedrock-runtime", region_name=self.region_name, **filtered_creds )
        self.inference_parameters = [
            "maxTokens",
            "temperature",
            "topP",
            "stopSequences",
        ]

    def normalize_response(self, response):
        """Normalize the response from the Bedrock API to match OpenAI's response format."""
        norm_response = ChatCompletionResponse()

        model_response = json.loads(response["body"].read())

        # Extract and print the response text.
        response_text = model_response["content"][0]["text"]
        norm_response.choices[0].message.content = response_text

        return norm_response

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Anthropic will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        system_message = []
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]

        formatted_messages = []

        for message in messages:
            # Skip system messages except the first one
            if message["role"] != "system":
                # Explicitly set both type and text fields
                content_item = {
                    "type": "text",
                    "text": message["content"]
                }
                formatted_messages.append({
                    "role": message["role"],
                    "content": [content_item]  # Wrap the content item in a list
                })

        # Maintain a list of Inference Parameters which Bedrock supports.
        # These fields need to be passed using inferenceConfig.
        # Rest all other fields are passed as additionalModelRequestFields.
        inference_config = {}
        additional_model_request_fields = {}

        # Iterate over the kwargs and separate the inference parameters and additional model request fields.
        for key, value in kwargs.items():
            if key in self.inference_parameters:
                inference_config[key] = value
            else:
                additional_model_request_fields[key] = value

        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0,
            "messages": formatted_messages,
            "system": system_message,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)
        # Call the Bedrock Converse API.
        response = self.client.invoke_model(modelId=model, body=request)

        return self.normalize_response(response)
