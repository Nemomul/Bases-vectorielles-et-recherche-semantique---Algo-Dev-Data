- Pourquoi KD-Tree échoue en dimension 768 ?
En haute dimension, tous les points deviennent "équidistants". Un KD-Tree doit explorer presque tous les nœuds → aussi lent qu'une recherche brute.

- HNSW — le graphe multi-couches
Imaginez un réseau routier :

Couche 2 = autoroutes (peu de nœuds, grands sauts)
Couche 1 = nationales
Couche 0 = toutes les rues (graphe complet)