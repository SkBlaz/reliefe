rm -rf _build
sphinx-apidoc -f -o source ../reliefe;
cp source/* .;
make html;
