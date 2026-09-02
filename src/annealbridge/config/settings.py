"""Server settings read from the environment (Phase 2 spec §9).

Composition root only: just ``annealbridge.interfaces.*`` may import this
package (spec §4). D-Wave credentials deliberately stay out of these
settings — Ocean's native config handles them (spec §18).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from annealbridge.orchestration.policy import ExecutionPolicy


class ServerSettings(BaseSettings):
    """Environment-driven configuration with the ``ANNEALBRIDGE_`` prefix."""

    model_config = SettingsConfigDict(env_prefix="ANNEALBRIDGE_")

    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = 24
    max_qpu_reads: int = 1000
    max_qpu_annealing_time_us: float = 2000.0
    max_remote_time_seconds: int = 300
    max_concurrent_solves: int = 4
    http_host: str = "127.0.0.1"
    http_port: int = 8000

    def to_policy(self) -> ExecutionPolicy:
        """Build the :class:`ExecutionPolicy` these settings describe."""
        return ExecutionPolicy(
            allow_remote=self.allow_remote,
            allow_remote_retries=self.allow_remote_retries,
            exact_max_variables=self.exact_max_variables,
            max_qpu_reads=self.max_qpu_reads,
            max_qpu_annealing_time_us=self.max_qpu_annealing_time_us,
            max_remote_time_seconds=self.max_remote_time_seconds,
            max_concurrent_solves=self.max_concurrent_solves,
        )
