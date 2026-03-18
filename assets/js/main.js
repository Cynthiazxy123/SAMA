const buttons = document.querySelectorAll("[data-copy-target]");

buttons.forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent);
      const original = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = original;
      }, 1200);
    } catch (error) {
      button.textContent = "Copy failed";
    }
  });
});

const syncGroups = new Map();

document.querySelectorAll("video[data-sync-group]").forEach((video) => {
  const groupId = video.dataset.syncGroup;
  if (!syncGroups.has(groupId)) {
    syncGroups.set(groupId, []);
  }
  syncGroups.get(groupId).push(video);
});

const syncTime = (source, target) => {
  if (Math.abs(source.currentTime - target.currentTime) > 0.12) {
    target.currentTime = source.currentTime;
  }
};

syncGroups.forEach((videos) => {
  if (videos.length < 2) return;

  let isSyncing = false;

  const syncOthers = (source, callback) => {
    if (isSyncing) return;
    isSyncing = true;
    try {
      videos.forEach((target) => {
        if (target === source) return;
        callback(target);
      });
    } finally {
      isSyncing = false;
    }
  };

  const bindSync = (source) => {
    source.addEventListener("play", async () => {
      syncOthers(source, (target) => {
        syncTime(source, target);
        target.playbackRate = source.playbackRate;
        if (target.paused) {
          target.play().catch(() => {
            // Ignore autoplay restrictions; manual interaction will still sync.
          });
        }
      });
    });

    source.addEventListener("playing", () => {
      syncOthers(source, (target) => {
        syncTime(source, target);
        target.playbackRate = source.playbackRate;
        if (target.paused) {
          target.play().catch(() => {
            // Ignore autoplay restrictions; manual interaction will still sync.
          });
        }
      });
    });

    source.addEventListener("pause", () => {
      syncOthers(source, (target) => {
        if (!target.paused) {
          target.pause();
        }
      });
    });

    source.addEventListener("seeking", () => {
      syncOthers(source, (target) => {
        syncTime(source, target);
      });
    });

    source.addEventListener("timeupdate", () => {
      syncOthers(source, (target) => {
        syncTime(source, target);
      });
    });

    source.addEventListener("ratechange", () => {
      syncOthers(source, (target) => {
        target.playbackRate = source.playbackRate;
      });
    });

    source.addEventListener("ended", () => {
      syncOthers(source, (target) => {
        target.pause();
        target.currentTime = source.currentTime;
      });
    });
  };

  videos.forEach((video) => {
    bindSync(video);
  });

  videos.forEach((video) => {
    video.muted = true;
  });
});
