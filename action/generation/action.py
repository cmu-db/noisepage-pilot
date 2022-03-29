from abc import ABC, abstractmethod


class Action(ABC):
    @abstractmethod
    def _to_sql(self) -> str:
        raise NotImplementedError("Should be implemented by child classes")

    def __str__(self):
        return self._to_sql()


class ActionGenerator(ABC):
    # @abstractmethod
    # def __len__(self) -> int:
    #     raise NotImplementedError("Should be implemented by child classes")

    # @abstractmethod
    # def __getitem__(self, action_ind) -> str:
    #     raise NotImplementedError("Should be implemented by child classes")

    @abstractmethod
    def __iter__(self, action_ind) -> str:
        raise NotImplementedError("Should be implemented by child classes")
