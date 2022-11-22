# Engineering-practices-ml

## Установка пакетного менеджера (poetry)

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

## Развёртывание окружения

```shell
poetry install
```

## Сборка пакета

```shell
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi <PYPI-TOKEN>
poetry publish --build --repository test-pypi
```

## Ссылка на пакет в pypi-test

```
https://test.pypi.org/project/decision-tree-classifier/
```

## Установка пакета из pypi-test

```shell
poetry source add test-pypi https://test.pypi.org/simple/
poetry add --source test-pypi decision-tree-classifier
```

## Запуск 
```shell
cd src
python3 main.py
```