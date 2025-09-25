class CustomException(Exception):
    """
    Custom exception class for Flipkart Product Recommendation project.
    Wraps original exceptions with a context message.
    """

    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

    def __str__(self):
        if self.original_exception:
            return f"{self.args[0]} | Original exception: {repr(self.original_exception)}"
        return self.args[0]
