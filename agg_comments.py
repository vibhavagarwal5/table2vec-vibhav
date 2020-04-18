import pathlib
import os
import toml

def get_comments(output_dir):
    path = pathlib.Path(output_dir)
    
    comments = []
    
    for conf_path in path.rglob("config.toml"):
        try:
            comment = toml.load(conf_path.resolve())["comment"]
        except:
            comment = ''
        comments.append((conf_path.parent.name, comment))
    
    
    comments.sort(key=lambda x: list(map(int, x[0].split("_"))))
    
    for name, comment in comments:
        print(f"{name}: {comment}\n")

print(get_comments("./output"))
