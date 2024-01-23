class TerminalColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def blue(self, string: str) -> str:
        return self.OKBLUE + string + self.ENDC

    def cyan(self, string: str) -> str:
        return self.OKCYAN + string + self.ENDC

    def green(self, string: str) -> str:
        return self.OKGREEN + string + self.ENDC

    def warning(self, string: str) -> str:
        return self.WARNING + string + self.ENDC

    def fail(self, string: str) -> str:
        return self.FAIL + string + self.ENDC

    def bold(self, string: str) -> str:
        return self.FAIL + string + self.ENDC

    def underline(self, string: str) -> str:
        return self.FAIL + string + self.ENDC
