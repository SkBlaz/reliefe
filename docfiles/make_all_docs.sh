rm -rf _build
sphinx-apidoc -f -o source ../reliefe;
make html;
