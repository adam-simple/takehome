import dspy


class Together(dspy.LM):
    def __init__(
        self,
        model,
        api_base="",
        api_key="",
        **kwargs,
    ):
        model = f"together_ai/{model}"

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
