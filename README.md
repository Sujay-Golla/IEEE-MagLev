# ieee-maglev
## directories
- `part1/` is instantanious levitation of a platform using the staggered coil configuration
- `part2/` is a continuation of part1 but over a duration of time
- `part3/` is a continuation of part2 with the addition of translations of the platform in the x y directions

## results

## usage
```sh
git clone https://github.com/Sujay-Golla/IEEE-MagLev.git

git checkout jim

cd part1 # or another part

# RUNNING A SIMULATION
python main.py

# RUNNING A SIMULATION (with logs)
python main.py > results/"$(date +%Y-%m-%d_%H-%M-%S).txt"

```
