# GenDiL Migration Scripts

These scripts apply conservative, mechanical source migrations between GenDiL
versions. They are intended for simple renames of public symbols and include
paths; they do not parse C++ or resolve semantic API changes.

Every script is a dry run by default:

```sh
scripts/migration/v0.0.4tov0.0.5.sh /path/to/project
```

Pass `--apply` to edit files in place:

```sh
scripts/migration/v0.0.4tov0.0.5.sh --apply /path/to/project
```

Useful options:

- `--check` exits with a nonzero status if automatic replacements are still
  needed.
- `--backup` creates a `.bak` copy before the first edit to each file when used
  with `--apply`.
- `--include-docs` also scans `.md`, `.rst`, and `.txt` files.

To migrate code from `v0.0.3` to the planned `v0.0.5` API, run the scripts in
order:

```sh
scripts/migration/v0.0.3tov0.0.4.sh --apply /path/to/project
scripts/migration/v0.0.4tov0.0.5.sh --apply /path/to/project
```
