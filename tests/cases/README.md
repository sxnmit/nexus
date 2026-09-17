# Regression cases

One JSON file per recorded reply, exported by the reviewer into a brief and
saved here by whoever builds the fix. Each holds what the agent saw -- the
earlier conversation, the message, the open tasks as the tools fetched them --
and what it did, with the judge's grade and the owner's label at the time.

    python evals.py                      # replay every case here, grade each again
    python evals.py tests/cases/abc.json # just these

A replay calls the real model and the real judge, so it is not part of the
unit tests. A fix should turn the cases that motivated it green without
turning any older one red.
