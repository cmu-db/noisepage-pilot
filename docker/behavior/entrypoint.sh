#!/bin/bash

set -e

function main() {
    if [[ $# -eq 0 ]]; then
        printf "\nHello, World!\n"
    else
        printf "\nHello, %s!\n" "${1}"
    fi
}

main "$@"

/bin/bash