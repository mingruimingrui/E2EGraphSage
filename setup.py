import os
import setuptools


def form_console_script(script_name):
    return 'e2e_{basename}=e2e_graphsage.bin.{basename}:main'.format(
        basename=os.path.splitext(script_name)[0])


script_names = os.listdir('e2e_graphsage/bin')
script_names = filter(lambda f: f.endswith('.py'), script_names)
script_names = filter(lambda f: f != '__init__.py', script_names)
console_scripts = [form_console_script(n) for n in script_names]
exclude_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d != 'gmlp']

print('The following console scripts will be added')
for script in console_scripts:
    print('- {}'.format(script))
print()  # Add new line

setuptools.setup(
    name='E2EGraphSage',
    version='0.1b',
    packages=setuptools.find_packages(exclude=exclude_dirs),
    entry_points={'console_scripts': console_scripts}
)
