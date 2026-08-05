from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptCall:
    system_prompt: str
    user_prompt: str


@dataclass
class FakeProvider:
    response: str
    calls: list[PromptCall] = field(default_factory=list, init=False)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            PromptCall(system_prompt=system_prompt, user_prompt=user_prompt)
        )
        return self.response
