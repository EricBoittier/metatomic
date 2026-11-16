# Changelog

All notable changes to metatomic-core are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatensor/)

<!-- Possible sections

#### Added

#### Fixed

#### Changed

#### Removed
-->

### Fixed

- Validate the samples of a model's outputs by looking up each entry, instead
  of building the union and intersection of labels. The outputs of a model
  running on an accelerator can now be checked against systems and
  `selected_atoms` that live on another device.
