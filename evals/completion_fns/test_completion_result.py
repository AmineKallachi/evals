# test_completion_result.py

from evals.api import CompletionResult


class MyRetrievalCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]

# Test Case 1: Creating an instance of MyRetrievalCompletionResult
response_text = "This is a test response."
completion_result = MyRetrievalCompletionResult(response_text)

# Test Case 2: Testing get_completions() method
completions = completion_result.get_completions()
print("Completions:", completions)  # Output: Completions: ['This is a test response.']

# Test Case 3: Testing the response attribute
print("Response:", completion_result.response)  # Output: Response: This is a test response.
