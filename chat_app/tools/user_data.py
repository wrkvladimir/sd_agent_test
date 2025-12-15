from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class UserData:
    name: str
    age: int


_FIRST_NAMES = [
    "Иван",
    "Алексей",
    "Сергей",
    "Дмитрий",
    "Ольга",
    "Павел",
]

_LAST_NAMES = [
    "Иванов",
    "Петров",
    "Сидоров",
    "Смирнов",
    "Кузнецов",
    "Васильев",
    "Морозов",
]


def get_user_data() -> UserData:
    """
    Stub implementation of get_user_data tool.

    Для первого сообщения в диалоге возвращает случайное русское ФИО и возраст
    от 18 до 120 лет.
    """
    first = random.choice(_FIRST_NAMES)
    last = random.choice(_LAST_NAMES)
    # Формируем простое "Фамилия Имя".
    name = f"{last} {first}"
    age = random.randint(18, 120)
    return UserData(name=name, age=age)
