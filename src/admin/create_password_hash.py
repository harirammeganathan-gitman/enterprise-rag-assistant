"""Prompt for an admin password and print its Argon2id hash."""

from getpass import getpass

from argon2 import PasswordHasher


def main() -> None:
    password = getpass("Admin password: ")
    confirmation = getpass("Confirm admin password: ")
    if not password:
        raise SystemExit("Password must not be empty.")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")
    print(PasswordHasher().hash(password))


if __name__ == "__main__":
    main()
