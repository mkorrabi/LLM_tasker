# IHMs llm-tasker

## Classification fermée

> Guide d'utilisation de l'interface de classification par LLM

Ce document décrit les étapes pour utiliser l'interface de classification fermée par LLM. Suivez ces instructions pour configurer et utiliser le système efficacement.

### Première utilisation

Lors de la première utilisation de l'interface, vous devrez configurer les paramètres de classification.

#### Choix du modèle et des paramètres avancés

- **Modèle et Paramètres avancés** : Les paramètres avancés sont pré-réglés sur les valeurs idéales pour la classification.

#### Configuration du prompt

1. **Instructions** : Définissez les règles que le modèle doit suivre.
2. **Classes** : Définissez les classes à utiliser ainsi que leurs définitions.
3. **Exemples** :
   - **Exemple et Classe** : Définissez des exemples que le LLM utilisera pour améliorer sa compréhension et sa réponse.

#### Paramètres avancés du prompt

- **Système de prompt** : Paramètre pour l'utilisation du prompt dans le système et non en mode utilisateur.
- **L'input est dans le prompt** : Assurez-vous que `{input}` est inclus dans les instructions.
- **Utiliser le rôle d'AI pour les exemples** : Simule une interaction avec le LLM plutôt que d'utiliser directement les exemples dans le prompt.

Après avoir renseigné tous les paramètres, vous pouvez télécharger le JSON englobant tous les paramètres via le bouton **Télécharger les paramètres sauvegardés en json**.

### Utilisation ultérieure

Si vous avez déjà configuré vos paramètres lors d'une utilisation précédente et enregistré ces paramètres dans un fichier JSON :

- **Chargement du fichier JSON** : Récupérez vos paramètres depuis votre stockage local en utilisant le bouton **Charger fichier de paramétrage JSON**.

### Méthodes de classification

- **Classification avec un input texte** : Permet une classification simple avec un input textuel.
- **Classification avec un fichier en input** : Utilisez un fichier Excel qui contient une colonne `<inputs>` placée en première position pour réaliser la classification.
