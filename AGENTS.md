# Repository Guidance

  * Treat `trade_data["symbol"]` as an opaque journal identifier.
  * Do not propose normalization or format validation for symbols unless the
    repository already enforces that rule or the user explicitly asks for it.
  * Only recommend symbol-related changes when there is concrete repository
    evidence and a clear practical benefit.
