"""Auth commands: set, list, remove secrets in the encrypted vault."""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.table import Table

from exo_mcp_cli.output import console, print_error, print_success
from exo_mcp_cli.vault import VaultError

auth_app = typer.Typer(
    name="auth",
    help="Manage encrypted credentials.",
    no_args_is_help=True,
)


@auth_app.command("set")
def auth_set(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Secret name (used in ${vault:NAME} references).")],
    value: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Secret value to store. "
                "Omit to be prompted interactively (recommended — avoids shell history). "
                "Use --stdin to read from standard input."
            )
        ),
    ] = None,
    stdin: Annotated[
        bool,
        typer.Option("--stdin", help="Read the secret value from stdin (for scripting)."),
    ] = False,
) -> None:
    """Store a secret in the encrypted vault.

    The secret value is read from an interactive hidden prompt by default so
    that it never appears in shell history or process listings.  Pass the value
    as a positional argument only when you are sure the environment is safe
    (e.g. inside CI where history is not persisted), or use --stdin to pipe it
    in from a file / password-manager.
    """
    from exo_mcp_cli.main import get_vault

    # Resolve value: explicit arg > --stdin > interactive prompt
    if value is not None:
        secret = value
    elif stdin:
        secret = sys.stdin.readline().rstrip("\n")
    else:
        secret = typer.prompt(f"Value for '{name}'", hide_input=True, confirmation_prompt=True)

    try:
        vault = get_vault(ctx)
        vault.set(name, secret)
    except VaultError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    print_success(f"Secret '{name}' stored in vault.")
    console.print(f"[dim]Use ${{vault:{name}}} in mcp.json to reference it.[/dim]")


@auth_app.command("list")
def auth_list(ctx: typer.Context) -> None:
    """List all secret names in the vault (values are not shown)."""
    from exo_mcp_cli.main import get_vault

    try:
        vault = get_vault(ctx)
        names = vault.list_names()
    except VaultError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    if not names:
        console.print("[dim]No secrets stored in vault.[/dim]")
        return

    table = Table(title="Vault Secrets")
    table.add_column("Name", style="cyan")
    table.add_column("Reference", style="dim")
    for n in names:
        table.add_row(n, f"${{vault:{n}}}")
    console.print(table)


@auth_app.command("remove")
def auth_remove(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Secret name to remove.")],
) -> None:
    """Remove a secret from the vault."""
    from exo_mcp_cli.main import get_vault

    try:
        vault = get_vault(ctx)
        if not vault.remove(name):
            print_error(f"Secret '{name}' not found in vault.")
            raise typer.Exit(code=1)
    except VaultError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    print_success(f"Secret '{name}' removed from vault.")
