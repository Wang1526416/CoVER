class DummyParse:

    @staticmethod
    def parse(response: str, *args, **kwargs) -> dict:
        return response.strip()
