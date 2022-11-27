# Engineering-practices-ml

----

## Оглавление

1. [Установка и запуск](#Установка-и-запуск)
2. [СodeStyle](#CodeStyle)

____

## Установка и запуск

## Установка пакетного менеджера (poetry)

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

## Развёртывание dev окружения

```shell
poetry install
```

## Развёртывание prod окружения

```shell
poetry install --without dev
```

## Сборка пакета

```shell
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi <PYPI-TOKEN>
poetry publish --build --repository test-pypi
```

## Ссылка на пакет в pypi-test

```
https://test.pypi.org/project/hse-decision-tree-classifier/
```

## Установка пакета из pypi-test

```shell
poetry source add test-pypi https://test.pypi.org/simple/
poetry add --source test-pypi hse-decision-tree-classifier
```

## Запуск

```shell
cd decision-tree-classifier
python3.9 main.py
```

____

## CodeStyle

Используемые инструменты:
* Форматирование: [black](https://github.com/psf/black) + [isort](https://github.com/PyCQA/isort)
* Линтеры: PyFlakes и pycodestyle ([flake8](https://github.com/PyCQA/flake8))
* Плагины для flake8:
  * [flake8-simplify](https://github.com/MartinThoma/flake8-simplify)
  * [flake8-return](https://github.com/afonasev/flake8-return)
  * [pep8-naming](https://github.com/PyCQA/pep8-naming)
  * [flake8-variables-names](https://github.com/best-doctor/flake8-variables-names)
  * [flake8-docstrings](https://github.com/pycqa/flake8-docstrings)
  * [flake8-unused-arguments](https://github.com/nhoad/flake8-unused-arguments)

Для прогона линтеров и форматирования кода при коммите или пуше в удалённый репозиторий установите pre-commit:

```shell
pre-commit install
```

Зафиксированные проблемы flake8 в файле linting.md. Обновить вручную:

```shell
flake8 decision-tree-classifier/ | sed 's/^/* /g' | sed -z 's/\n/\n\n/g' > linting.md
````
