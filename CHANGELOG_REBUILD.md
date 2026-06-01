# Rebuild changelog

## Structural changes

- Created package layout with `src/himars`.
- Split monolithic notebooks into modules: data loading, ICF, objectives, operators, Pareto utilities, algorithms, metrics, selection, and experiment runner.
- Added YAML configs for MovieLens and ModCloth.
- Added tests and CLI scripts.

## Corrections

1. **Adjusted cosine similarity**
   - The paper says ICF uses adjusted cosine similarity. The main notebooks used plain cosine on zero-filled ratings. The rebuild defaults to adjusted cosine.

2. **Weighted-sum denominator**
   - The paper formula uses absolute similarities in the denominator. The rebuild uses `sum(abs(similarity))`.

3. **HANI naming**
   - Old notebook labels `HANv3` and `HANv4` are replaced by `HANIv1` and `HANIv2`.

4. **HANIv2 mutation bug**
   - The notebook implementation of `HANv4` computed crossover offspring `CT` but updated the population with `Ct` without assigning it in the loop. The rebuild explicitly computes mutated offspring before updating the population.

5. **Random seeds**
   - Randomness now flows through `numpy.random.Generator`.

6. **Numerical safety**
   - Zero objective ranges, empty fronts, and one-point fronts are handled without runtime warnings.

7. **No global state**
   - User ID, train/test data, similarity matrix, candidate lists, and parameters are passed explicitly.

## Remaining assumptions

- The exact ModCloth column names must be verified against the real CSV.
- Exact numerical agreement with the manuscript requires the original datasets and the old saved simulation outputs.
- Pareto-front metric definitions in the manuscript/code are not fully standard; this rebuild follows the notebook direction but guards edge cases.
