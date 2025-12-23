# Dev Env 


# To edit file created on container:
sudo chown -R $USER:$(id -gn) ~/gpu-workspace

# Mount dev container on terminal
devcontainer up --workspace-folder .
devcontainer up --workspace-folder . --log-level debug

devcontainer exec --workspace-folder . /bin/zsh -li
devcontainer exec --workspace-folder . /bin/bash


# ToDo
1. Two way edit ... auto user mound (?)
2. mount .ssh, bashrc, docker, HF ... all config
3. Install important htop, etc ...
4. AI tools, gemini CLI (auth)? ... Nah