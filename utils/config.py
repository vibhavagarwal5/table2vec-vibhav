import toml


class Config:
    # path to default configuration file
    default = "./config.toml"

    def __init__(self):
        """Load and save configuration files and options."""
        # parse default config
        self.opt = toml.load(self.default)
        self.parsed = [self.default]

    def load(self, path):
        """Load all configuration options from given file. Successive calls
        override previously read options. Base options are provided by default
        configuration file.

        :param path: Path to custom configuration file.
        """
        opt = toml.load(path)
        self.opt = {**self.opt, **opt}
        self.parsed.append(path)

    def save(self, path):
        """Save configuration options at path. File will be created if it
        doesn't exist or overwritten if it exists.

        :param path: Path to file where configuration options are saved.
        """
        with open(path, "w") as f:
            toml.dump(self.opt, f)

    def add_to_config(self, key, value):
        """
        Add key, value pair to config. Overrides if already exists.

        :param key: key in the hash table
        :param value: value corresponding to key in the hash table
        """
        self.opt[key] = value

    def __getitem__(self, name):
        return self.opt[name]
