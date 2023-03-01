
# SNN

## **SPIKING NEURAL NETWORKS(SNN)**

> **Les réseaux de neurones à pointes ( SNN )** sont des réseaux de neurones artificiels qui imitent plus étroitement les réseaux de neurones naturels. En plus de l'état neuronal et synaptique, les SNN intègrent le concept de temps dans leur modèle de fonctionnement. L'idée est que les neurones du SNN ne transmettent pas d'informations à chaque cycle de propagation (comme cela se produit avec les réseaux de perceptrons multicouches typiques), mais transmettent plutôt des informations uniquement lorsqu'un potentiel de membrane -
>   une qualité intrinsèque du neurone liée à sa membrane électrique
> charge – atteint une valeur spécifique, appelée le seuil. Quand le
> le potentiel de membrane atteint le seuil, le neurone se déclenche et
> génère un signal qui se déplace vers d'autres neurones qui, à leur tour,
> augmentent ou diminuent leurs potentiels en réponse à ce signal. UN modèle de neurone qui se déclenche au moment du franchissement du seuil est également appelé un modèle de neurone à pointes.
>
> Le modèle de neurones à pointes le plus important est le modèle d'intégration et de déclenchement qui fuit. Dans le modèle d'intégration et de tir, le niveau d'activation momentané (modélisé comme une équation différentielle)
>   est normalement considéré comme l'état du neurone, avec des pointes entrantes
> poussant cette valeur vers le haut ou vers le bas, jusqu'à ce que l'état finisse par soit
> se désintègre ou - si le seuil de déclenchement est atteint - le neurone se déclenche.
> Après le déclenchement, la variable d'état est réinitialisée à une valeur inférieure.
>
> Diverses méthodes de décodage existent pour interpréter le train de pointes  sortant sous la forme d'un nombre à valeur réelle, en s'appuyant soit sur la fréquence des pointes (code de débit), soit sur le temps jusqu'au premier pic après stimulation, soit sur l'intervalle entre les pointes.

---

## **FONDEMENTS**

> Les informations dans le cerveau sont représentées sous forme de potentiels d'action (pics de neurones), qui peuvent être regroupés en trains de pics ou même en vagues coordonnées d'activité cérébrale. Une question fondamentale des neurosciences est de déterminer si les neurones communiquent par un rythme ou un code temporel .  Le codage temporel suggère qu'un seul neurone à pointes peut remplacer des centaines d'unités cachées sur un réseau neuronal sigmoïde .
>
> Les informations dans le cerveau sont représentées sous forme de potentiels d'action (pics de neurones), qui peuvent être regroupés en trains de pics ou même en vagues coordonnées d'activité cérébrale. Une question fondamentale des neurosciences est de déterminer si les neurones communiquent par un rythme ou un code temporel .  Le codage temporel suggère qu'un seul neurone à pointes peut remplacer des centaines d'unités cachées sur un réseau neuronal sigmoïde .
>
> Dans un réseau de neurones à pointes, l'état actuel d'un neurone est défini comme son potentiel de membrane (éventuellement modélisé par une équation différentielle). Une impulsion d'entrée fait monter le potentiel de membrane pendant un certain temps, puis diminue progressivement. Des schémas de codage ont été construits pour interpréter ces séquences d'impulsions comme un nombre, en tenant compte à la fois de la fréquence des impulsions et de l'intervalle des impulsions. Un modèle de réseau neuronal basé sur le temps de génération d'impulsions peut être établi. En utilisant l'heure exacte d'apparition des impulsions, un réseau de neurones peut utiliser plus d'informations et offrir de meilleures propriétés de calcul.
>
> L'approche SNN produit une sortie continue au lieu de la sortie binaire des ANN traditionnels. Les trains d'impulsions ne sont pas facilement interprétables, d'où la nécessité de schémas de codage comme ci-dessus. Cependant, une représentation de train d'impulsions peut être plus adaptée au traitement de données spatio-temporelles (ou à la classification continue des données sensorielles du monde réel).  Les SNN considèrent l'espace en connectant les neurones uniquement aux neurones voisins afin qu'ils traitent les blocs d'entrée séparément (similaire au CNN utilisant des filtres). Ils considèrent le temps en encodant les informations sous forme de trains d'impulsions afin de ne pas perdre d'informations dans un codage binaire. Cela évite la complexité supplémentaire d'un réseau de neurones récurrent(RNN). Il s'avère que les neurones à impulsions sont des unités de calcul plus puissantes que les neurones artificiels traditionnels.
>
> Les SNN sont théoriquement plus puissants que les réseaux de seconde génération ; cependant, les problèmes de formation SNN et les exigences matérielles limitent leur utilisation. Bien que des méthodes d'apprentissage biologiquement inspirées non supervisées soient disponibles telles que l&#39;apprentissage Hebbian et le STDP , aucune méthode de formation supervisée efficace n'est adaptée aux SNN qui peuvent fournir de meilleures performances que les réseaux de deuxième génération.  L'activation basée sur les pointes des SNN n'est pas différentiable, ce qui rend difficile le développement de méthodes d'entraînement basées sur la descente de gradient pour effectuer la rétropropagation des erreurs, bien que quelques algorithmes récents tels que NormAD  et NormAD multicouche aient démontré de bonnes performances d'entraînement grâce à une approximation appropriée du gradient d'activation basée sur les pointes.
>
> Les SNN ont des coûts de calcul beaucoup plus importants pour simuler des modèles neuronaux réalistes que les ANN traditionnels
>
> Les réseaux de neurones à couplage d&#39;impulsions (PCNN) sont souvent confondus avec les SNN. Un PCNN peut être considéré comme une sorte de SNN.
>
> Actuellement, il existe quelques défis lors de l'utilisation des SNN sur lesquels les chercheurs travaillent activement. Le premier défi concerne la non-différenciabilité de la non-linéarité de pointe. Les expressions des méthodes d'apprentissage vers l'avant et vers l'arrière contiennent la dérivée de la fonction d'activation neurale qui n'est pas différentiable car la sortie du neurone est soit 1 lorsqu'elle pointe, soit 0 sinon. Ce comportement tout ou rien de la non-linéarité de pointe binaire empêche les gradients de « couler » et rend les neurones LIF inadaptés à l'optimisation basée sur les gradients. Le deuxième défi concerne la mise en œuvre de l'algorithme d'optimisation lui-même. La BP standard peut être coûteuse en termes de calcul, de mémoire et de communication et peut être mal adaptée aux contraintes dictées par le matériel qui l'implémente (par exemple, un ordinateur, un cerveau, En ce qui concerne le premier défi, il y en a plusieurs approchés pour le surmonter. Quelques-uns d'entre eux sont :
>
> * recours à des règles d'apprentissage locales entièrement d'inspiration biologique pour les unités cachées
> * traduire les NN "basés sur le débit" formés de manière conventionnelle en SNN
>
> * lisser le modèle de réseau pour qu'il soit continuellement différentiable
> * définir un SG (Surogate Gradient) comme une relaxation continue des gradients réels

## **APPLICATIONS**

> Les SNN peuvent en principe s'appliquer aux mêmes applications que les ANN traditionnels. De plus, les SNN peuvent modéliser le système nerveux central d'organismes biologiques, comme un insecte cherchant de la nourriture sans connaissance préalable de l'environnement.  En raison de leur réalisme relatif, ils peuvent être utilisés pour étudier le fonctionnement des circuits neuronaux biologiques . En partant d'une hypothèse sur la topologie d'un circuit neuronal biologique et sa fonction, les enregistrements de ce circuit peuvent être comparés à la sortie du SNN correspondant, évaluant la plausibilité de l'hypothèse. Cependant, il existe un manque de mécanismes de formation efficaces pour les SNN, qui peuvent être inhibiteurs pour certaines applications, y compris les tâches de vision par ordinateur.

>     À partir de 2019, les SNN sont à la traîne des ANN en termes de précision, mais l'écart diminue et a disparu pour certaines tâches.

>
>     Lors de l'utilisation de SNN pour des données basées sur des images, nous devons convertir des images statiques en codage de trains de pointes binaires. 

> **Types d'encodages :**

* >   Le codage temporel génère un pic par neurone dans lequel la latence du pic est inversement proportionnelle à l'intensité du pixel.

* > Le codage de débit convertit l'intensité des pixels en un train de pointes où le nombre de pointes est proportionnel à l'intensité des pixels.
* > Le codage direct utilise une couche entraînable pour générer une valeur flottante pour chaque pas de temps. Nous avons une couche apprenable qui convertit chaque pixel à un certain pas de temps en nombre flottant, puis le seuil est utilisé sur les nombres flottants générés pour voir s'ils seront 1 ou 0.
* > Le codage de phase code les informations temporelles dans des modèles de pointes basés sur un oscillateur global.
* > Le codage en rafale transmet la rafale de pointes dans une courte durée, augmentant la fiabilité de la communication synaptique entre les neurones.
  >
