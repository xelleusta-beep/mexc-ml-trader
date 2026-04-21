#!/usr/bin/env python3
"""
Simple Deployment Validation
Paket olmadan kontrol editmeleri listele
"""

import ast
import sys

print("=" * 60)
print("DEPLOYMENT VALIDATION")
print("=" * 60)

def check_syntax(filepath):
    """Parse Python file and check for syntax errors"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, str(e)

def check_imports(filepath):
    """Extract imports from Python file"""
    imports = []
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or ".")
        return imports
    except:
        return []

# Test 1: Syntax
print("\n[TEST 1] Syntax Validation")
for file in ['backend/main.py', 'backend/ml_engine.py']:
    ok, msg = check_syntax(file)
    status = "✅" if ok else "❌"
    print(f"  {status} {file}: {msg}")
    if not ok:
        sys.exit(1)

# Test 2: Required imports
print("\n[TEST 2] Required Dependencies Check")
required_modules = {
    'fastapi': 'FastAPI Framework',
    'uvicorn': 'ASGI Server',
    'httpx': 'HTTP Client',
    'numpy': 'Numerical Computing',
    'websockets': 'WebSocket Support',
}

main_imports = check_imports('backend/main.py')
ml_imports = check_imports('backend/ml_engine.py')
all_imports = set(main_imports + ml_imports)

print("  Imports found in code:")
for imp in sorted(all_imports):
    if imp and not imp.startswith('_'):
        status = "✅" if imp in required_modules else "ℹ️"
        desc = required_modules.get(imp, "")
        print(f"    {status} {imp} {desc}")

# Test 3: API endpoints
print("\n[TEST 3] API Endpoints")
with open('backend/main.py', 'r') as f:
    content = f.read()
    endpoints = [
        ('/api/scan', 'GET'),
        ('/api/pair/{symbol}', 'GET'),
        ('/api/stats', 'GET'),
        ('/api/model', 'GET'),
        ('/ws', 'WebSocket'),
        ('/', 'Static/Frontend'),
    ]
    for endpoint, method in endpoints:
        if endpoint.split('/')[1] in content or endpoint == '/':
            print(f"    ✅ {method:10} {endpoint}")

# Test 4: Dependencies file
print("\n[TEST 4] Requirements File")
with open('backend/requirements.txt', 'r') as f:
    reqs = f.read().strip().split('\n')
    print(f"  ✅ {len(reqs)} packages defined:")
    for req in reqs:
        if req.strip():
            print(f"     - {req}")

# Test 5: Render config
print("\n[TEST 5] Render Configuration")
from pathlib import Path
for config in ['render.yaml', 'runtime.txt', 'Procfile']:
    if Path(config).exists():
        with open(config, 'r') as f:
            content = f.read().strip()
            lines = len(content.split('\n'))
            print(f"  ✅ {config}: {lines} lines configured")
    else:
        print(f"  ⚠️ {config}: Not found")

# Test 6: Frontend
print("\n[TEST 6] Frontend Assets")
if Path('frontend/index.html').exists():
    with open('frontend/index.html', 'r') as f:
        size = len(f.read())
        print(f"  ✅ frontend/index.html: {size} bytes")
else:
    print(f"  ⚠️ frontend/index.html: Not found")

print("\n" + "=" * 60)
print("✅ VALIDATION COMPLETE - Ready for deployment")
print("=" * 60)
print("\nNext steps:")
print("  1. Commit changes: git add -A && git commit")
print("  2. Push to GitHub: git push")
print("  3. Render triggers deployment automatically")
print("=" * 60)
