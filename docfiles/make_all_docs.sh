rm -rf _build
sphinx-apidoc -f -o source ../reliefe;
cp sources/* .;
make html;
