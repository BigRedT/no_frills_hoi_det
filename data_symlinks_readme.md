# Usage
Create symlinks to data in this directory. All paths in code will be specified relative to this directory.

Assuming all code is executed from the outermost project directory, define path to a file using
```Python
file_path = os.path.join(os.getcwd(),'data_symlink/<relative_path>')
```

# Symlink directories needed for HICO
- hico_clean
- hico_processed
- hico_exp