# theLab Execution in the HiveMind

Shared Docker lab environment for safe code execution.
Files are exchanged via `./the_wormhole` (mounted into the container as `/workspace`).

> Requires: `make awaken_hive`.

## Enable Execution

```python
from hivemind import HiveMind

hive = HiveMind(execute=True)
hive.add_drone("Unit734", "deepseek-r1:14b",
               "Execution-focused unit. Write and run code or shell commands. Minimal commentary.")
```

## Example

### Create a script in theLab

````python
hive.ask("""
@Unit734 Create a Python script in the_wormhole named 'analyze.py'
that prints the first 5 lines of any CSV file passed to it as an argument.

```python
import sys
import pandas as pd

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <filename>")
    raise SystemExit(1)

try:
    df = pd.read_csv(sys.argv[1])
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
```
""")
````

### Execute from theLab

````python
with open("the_wormhole/data.csv", "w") as f:
    f.write("id,value\n0,10\n1,20\n2,30\n3,40\n4,50\n")

hive.ask("""
@Unit734 Execute the 'analyze.py' script on 'data.csv'.

```sh
python analyze.py data.csv
```
""")
````
