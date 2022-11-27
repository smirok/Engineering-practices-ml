# Engineering-practices-ml

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