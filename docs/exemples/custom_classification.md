# Custom classification

Dans une classification _custom_, le format de sortir (`BaseItem.output`) est un objet `pydantic` défini par l'utilisateur.
On utilise le parser `LCPydanticOutputParser` défini dans le package. Il est nécessaire d'injecter une variable `{{json_shema_output}}` dans les instructions.

## Instructions

Les instructions dans cet exemple sont écrites dans un fichier au format _jinja_.

```jinja title="instructions"
--8<-- "prompts/custom_instruction.jinja"
```

## Script

```python linenums="1" title="Classification custom"
--8<-- "examples/tache_custom.py"
```
