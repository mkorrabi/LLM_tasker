"""Module d'implémentation d'un prompt avec langchain.

On définit des fonctionnalités à ajouter à la classe AbstractBasePrompt (pattern Mixin):
* LCPromptInputMixin: gérer l'input
* LCPromptExamplesMixin: gérer les examples
* LCPromptClassesMixin: gérer les classes
* LCPromptInstructionsMixin: gérer les instructions

On définit une classe flexibles pour gérer plusieurs tâches:
* LCPrompt

Pour ajouter une nouvelle classe, il faut qu'elle hérite de AbstractBasePrompt.

"""

from typing import Any, List, Union
from llmtasker.exceptions import LLMTaskerException
from llmtasker.prompts.base import AbstractBasePrompt
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage


class LCPromptInputMixin:
    """Fonctionnalités pour gérer la variable input."""

    def generate_input_with_user_role(self, **kwargs) -> HumanMessagePromptTemplate:
        """Retourne un message utilisateur avec la variable input.

        Cette fonctionnalité est le mode standard.
        L'input est ajouter comme un message utilisateur après le prompt system.

        Returns:
            HumanMessagePromptTemplate: message utilisateur au format langchain
        """
        # force f-string to avoid injection
        input_message = HumanMessagePromptTemplate.from_template(
            template="{input}",
            template_format="f-string",
        )
        return input_message


class LCPromptExamplesMixin:
    """Fonctionnalités pour gérer les examples dans le prompt."""

    def generate_examples_with_user_ai_roles(
        self, **kwargs
    ) -> List[Union[HumanMessagePromptTemplate, AIMessage]]:
        """Générer les exemples.

        Création de messages alternés avec le rôle User/AI
        pour donner des exemples de traitement de tâche au format: \
            - user: input
            - AI: output

        On utilise le mode chat pour forcer les exemples.

        Returns:
            List[Union[HumanMessagePromptTemplate, AIMessage]]: Liste de messages \
                au format langchain
        """
        examples_messages = []
        for item in self.examples:
            human_msg = HumanMessagePromptTemplate.from_template(
                template=item.format_input(),
                example=True,
                **kwargs,
            )

            ai_msg = AIMessage(content=item.format_output(), example=True)
            examples_messages.append(human_msg)
            examples_messages.append(ai_msg)
        return examples_messages

    def generate_examples_in_instructions_with_jinja(self, **kwargs):
        """Ajoute les examples dans une variables "examples" utilisable avec Jinja.

        Les exemples sont une liste exploitable avec un boucle dans le template jinja.
        """
        self.variables["examples"] = [i.format_as_example() for i in self.examples]
        return []

    def generate_examples_in_instructions_with_fstring(self, **kwargs):
        """Ajout les exemples dans une varible "examples" utilisable avec fstring.

        les exemples sont une chaine de caractère expoitable dans un fstring.
        Le format d'exemple est défini par la méthode format_as_example().
        """
        self.variables["examples"] = "\n".join(
            i.format_as_example() for i in self.examples
        )
        return []


class LCPromptClassesMixin:
    def generate_classes_in_instructions_with_jinja(self, **kwargs):
        if self.classes:
            self.variables["classes"] = [cl.format() for cl in self.classes]
        return []

    def generate_classes_in_instructions_with_fstring(self, **kwargs):
        if self.classes:
            self.variables["classes"] = "\n".join([cl.format() for cl in self.classes])
        return []


class LCPromptInstructionsMixin:
    def generate_instructions_with_system_role(self, **kwargs):
        if self.is_jinja_file(self.instructions):
            instruct_msg = SystemMessage(
                content=self.load_prompt_from_template(
                    self.instructions, self.variables
                )
            )
        else:
            instruct_msg = SystemMessagePromptTemplate.from_template(
                template=self.instructions,
                **kwargs,
            )
        return instruct_msg

    def generate_instructions_with_user_role(
        self,
        examples_messages: List[
            Union[HumanMessage, HumanMessagePromptTemplate, AIMessage]
        ] = [],
        **kwargs,
    ):
        if self.is_jinja_file(self.instructions):
            content = self.load_prompt_from_template(self.instructions, self.variables)
        else:
            content = self.instructions

        if len(examples_messages) > 0 and isinstance(
            examples_messages[0], (HumanMessage, HumanMessagePromptTemplate)
        ):
            if isinstance(examples_messages[0], HumanMessagePromptTemplate):
                first_example = examples_messages[0].prompt.template
            else:
                first_example = examples_messages[0].content
            content += "\n\n" + first_example
            del examples_messages[0]
        instruct_msg = HumanMessagePromptTemplate.from_template(
            template=content, **kwargs
        )
        return instruct_msg


class LCPrompt(
    AbstractBasePrompt,
    LCPromptInstructionsMixin,
    LCPromptInputMixin,
    LCPromptClassesMixin,
    LCPromptExamplesMixin,
):
    use_system_prompt: bool = True
    use_ai_user_role_for_examples: bool = False
    input_in_instructions: bool = False

    def generate(self, **kwargs: Any) -> ChatPromptTemplate:
        messages = []
        examples_messages = []
        instruct_msg = None
        message_to_analyze = None
        template_format = kwargs.get("template_format", "f-string")

        # Handle examples
        if self.examples:
            if self.use_ai_user_role_for_examples:
                examples_messages = self.generate_examples_with_user_ai_roles(**kwargs)
            else:
                if template_format == "f-string":
                    examples_messages = (
                        self.generate_examples_in_instructions_with_fstring(**kwargs)
                    )
                elif template_format == "jinja2":
                    examples_messages = (
                        self.generate_examples_in_instructions_with_jinja(**kwargs)
                    )
                else:
                    raise LLMTaskerException(
                        msg="template_format {template_format} doesn't exist.",
                        code="prompt_generate",
                    )

        # Handle classes
        # for now classes are in instructions throught variables
        if template_format == "f-string":
            self.generate_classes_in_instructions_with_fstring(**kwargs)
        elif template_format == "jinja2":
            self.generate_classes_in_instructions_with_jinja(**kwargs)

        # Handle instructions
        if self.use_system_prompt:
            instruct_msg = self.generate_instructions_with_system_role(**kwargs)
        else:
            instruct_msg = self.generate_instructions_with_user_role(
                examples_messages, **kwargs
            )

        # Handle input
        if self.input_in_instructions:
            self.is_var_in_instructions(instruct_msg, "input", template_format)
        else:
            message_to_analyze = self.generate_input_with_user_role(**kwargs)

        # Check every variables:
        # only work with f-string
        # with jinja, users can create loops like "{% for cl in classes %}"
        if template_format == "f-string":
            for var_name in self.variables:
                self.is_var_in_instructions(instruct_msg, var_name, template_format)

        # Organize all messages
        messages.append(instruct_msg)
        messages += examples_messages
        if message_to_analyze:
            messages.append(message_to_analyze)

        # return ChatPromptTemplate.from_messages(messages)
        return ChatPromptTemplate(messages=messages, partial_variables=self.variables)

    def is_var_in_instructions(
        self,
        instruction_msg: Union[BaseMessage, PromptTemplate],
        var_name: str = "classes",
        template_format: str = "f-string",
    ) -> bool:
        """Vérifie si la variable est dans les instructions.

        Si template_format == "f-string", on vérifie si {var_name} est dans les instructions.
        Si template_format == "jinja2", on vérifie si {{var_name}} est dans les instructions.

        Args:
            instruction_msg (Union[BaseMessage, PromptTemplate]): Instructions au format langchain
            var_name (str, optional): Nom de la variable à chercher. Defaults to "classes".

        Returns:
            bool: True si vrai, False sinon.
        """
        var_binding = (
            "{%s}" % var_name if template_format == "f-string" else "{{%s}}" % var_name
        )

        if isinstance(instruction_msg, BaseMessage):
            content = instruction_msg.content
        else:
            content = instruction_msg.prompt.template

        return self.check_input_var_in(content, var_binding)
