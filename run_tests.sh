#!/bin/bash
projname=$( basename $( pwd ) )
pytest --cov-report term-missing --cov="${projname}" tests/
