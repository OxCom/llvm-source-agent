# LLVM Source Agent

This project provides an AI agent that could answer on questions about the project in a ```source/``` directory.

## Run
1. Create folders for source code and index:
```bash
mkdir -p ./source
mkdir -p ./index
```

2. Clone the source code repository to the `source/` directory and remove .git directory to avoid indexing it:
```bash
git clone git@awesome/project.git ./source
rm -rf ./source/.git
```

3. Run the agent:
```bash
docker-compose up --build
```

