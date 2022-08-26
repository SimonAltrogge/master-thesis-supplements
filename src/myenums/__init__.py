from typing import TypeVar, TypeGuard
from enum import Enum


class HiddenValueEnum(Enum):
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}"


AnyOrderedEnum = TypeVar("AnyOrderedEnum", bound="OrderedEnum")


class OrderedEnum(Enum):
    def __ge__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value >= other.value

        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value > other.value

        return NotImplemented

    def __le__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value <= other.value

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value < other.value

        return NotImplemented

    def is_member_of_same_enum(
        self: AnyOrderedEnum, other: object
    ) -> TypeGuard[AnyOrderedEnum]:
        return self.__class__ is other.__class__
