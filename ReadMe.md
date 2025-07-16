### Initial Setup
if using the apis, add a local file named openrouter_key.txt
### Create Env (In alex cluster)

module load python

python3 -m venv emotions

source emotions/bin/activate

pip install -r requirements.txt

### Generate Dataset

python dataset_generator.py
