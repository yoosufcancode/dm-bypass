from pydantic import BaseModel


class IngestRequest(BaseModel):
    leagues: list[str]
    skip_download: bool = False


class IngestResult(BaseModel):
    features_paths: dict[str, str]   # league → path
    row_counts: dict[str, int]
    leagues_processed: list[str]
