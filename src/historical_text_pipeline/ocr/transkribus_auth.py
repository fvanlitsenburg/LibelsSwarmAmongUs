"""Authentication for the Transkribus processing API."""

from dataclasses import dataclass
from time import monotonic

import httpx
from pydantic import SecretStr

TOKEN_EXPIRY_MARGIN_SECONDS = 30


class TranskribusAuthenticationError(Exception):
    """Raised when a Transkribus access token cannot be obtained."""


@dataclass(slots=True)
class TokenState:
    """Cached Transkribus token information."""

    access_token: str
    refresh_token: str | None
    expires_at: float


class TranskribusAuthenticator:
    """Obtain and cache OpenID Connect access tokens."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        token_url: str,
        client_id: str,
        username: str,
        password: SecretStr,
    ) -> None:
        self._client = client
        self._token_url = token_url
        self._client_id = client_id
        self._username = username
        self._password = password
        self._token_state: TokenState | None = None

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing when needed."""

        if self._token_is_valid():
            assert self._token_state is not None
            return self._token_state.access_token

        if (
            self._token_state is not None
            and self._token_state.refresh_token is not None
        ):
            try:
                return self._refresh_access_token()
            except TranskribusAuthenticationError:
                self._token_state = None

        return self._authenticate_with_password()

    def _token_is_valid(self) -> bool:
        if self._token_state is None:
            return False

        return (
            monotonic() + TOKEN_EXPIRY_MARGIN_SECONDS
            < self._token_state.expires_at
        )

    def _authenticate_with_password(self) -> str:
        response = self._client.post(
            self._token_url,
            data={
                "grant_type": "password",
                "username": self._username,
                "password": self._password.get_secret_value(),
                "client_id": self._client_id,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        print(response.headers)
        return self._save_token_response(response)

    def _refresh_access_token(self) -> str:
        assert self._token_state is not None
        assert self._token_state.refresh_token is not None

        response = self._client.post(
            self._token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._token_state.refresh_token,
                "client_id": self._client_id,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        return self._save_token_response(response)

    def _save_token_response(
        self,
        response: httpx.Response,
    ) -> str:
        try:
            response.raise_for_status()
            payload = response.json()

            access_token = str(payload["access_token"])
            expires_in = float(payload.get("expires_in", 300))

            refresh_token_value = payload.get("refresh_token")
            refresh_token = (
                str(refresh_token_value)
                if refresh_token_value
                else None
            )

        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise TranskribusAuthenticationError(
                "Could not obtain a Transkribus access token."
            ) from error

        self._token_state = TokenState(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=monotonic() + expires_in,
        )

        return access_token