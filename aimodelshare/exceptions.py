class AuthorizationError(Exception):
    def __init__(self, error):
        Exception.__init__(self, error)


class AWSAccessError(Exception):
    def __init__(self, error):
        Exception.__init__(self, error)

class AWSUploadError(Exception):
    def __init__(self, error):
        Exception.__init__(self, error)
