# Waymax/WOSAC Transfer Hook

This repo does not implement a full Waymax/WOSAC baseline. The transfer hook exists to keep later comparison work structurally compatible.

## Purpose
The hook defines the minimum metadata and artifact interface needed to later test whether a decision-layer method developed in Scenario Dreamer transfers to a Waymax/WOSAC-style setup.

## Expected Inputs
- baseline or method `run_manifest.json`
- normalized `metrics.json`
- stable method identifier
- baseline checkpoint identity and environment set identity

## Expected Output
A small JSON request payload indicating:
- which run is being transferred
- what fixed baseline identity it depends on
- which future evaluator should consume it

## Non-Goals in V1
- no direct Waymax execution
- no WOSAC protobuf generation
- no official metrics integration here
