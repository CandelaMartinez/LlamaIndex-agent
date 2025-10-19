# diagnosticar_llamaindex.py
import llama_index.core

print(f"LlamaIndex version: {llama_index.core.__version__}")
print(f"LlamaIndex path: {llama_index.core.__file__}")

# Ver qué agentes están disponibles
import inspect
import llama_index.core.agent as agent_module

print("\nAvailable agents:")
for name in dir(agent_module):
    if 'Agent' in name and not name.startswith('_'):
        print(f"  - {name}")

# Ver métodos de ReActAgent si existe
if hasattr(agent_module, 'ReActAgent'):
    react_methods = [m for m in dir(agent_module.ReActAgent) if not m.startswith('_')]
    print(f"\nReActAgent methods: {react_methods}")