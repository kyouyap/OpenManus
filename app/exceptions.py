class ToolError(Exception):
    """ツールがエラーに遭遇した際に発生する例外です。"""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """OpenManusの全ての例外の基底クラスです"""


class TokenLimitExceeded(OpenManusError):
    """トークン制限を超過した際に発生する例外です"""
