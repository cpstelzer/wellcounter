# Git Troubleshooting: `origin/work` not found

When `git checkout -b work origin/work` fails with the error

```
fatal: 'origin/work' is not a commit and a branch 'work' cannot be created from it
```

the remote named `origin` does not currently advertise a branch called `work`. Git
cannot create a local branch that tracks a non-existent remote ref, so it aborts.

## Diagnose the missing branch

1. Verify the list of remote-tracking branches:
   ```bash
   git branch -r
   ```
   If `origin/work` is not in the list, the branch has not been fetched or has not been
   published to the remote server.
2. Refresh the remote references to make sure you have the latest branch list:
   ```bash
   git fetch origin --prune
   ```
   Re-run `git branch -r` to confirm whether `origin/work` now appears.
3. If the branch still does not show up, confirm its actual name on the server (for
   example, `origin/codex/work` or `origin/work-original`). Use `git ls-remote` if you
   do not have direct access to the hosting UI:
   ```bash
   git ls-remote --heads origin
   ```

## Create the local branch once the remote ref exists

* If you locate the correct remote branch, check it out directly:
  ```bash
  git checkout -b work origin/<actual-branch-name>
  ```
* If the branch truly has not been pushed, ask the author to publish it or push the
  commits yourself using:
  ```bash
  git push origin work
  ```

By confirming that the remote reference exists before creating your local tracking
branch, you avoid the `origin/work` lookup failure.
