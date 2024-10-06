document.addEventListener('DOMContentLoaded', function () {
  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      // Look for all instances of the Copy button
      const copyButtons = document.querySelectorAll(
        '.MuiStack-root .MuiButtonBase-root[aria-label="Copy"]'
      );

      copyButtons.forEach(function (copyButton) {
        // Hide each Copy button found
        if (copyButton) {
          copyButton.style.display = 'none';
        }
      });
    });
  });

  // Start observing the chat container for changes (adjust the selector if needed)
  const chatContainer = document.querySelector('#root');
  if (chatContainer) {
    observer.observe(chatContainer, { childList: true, subtree: true });
  }
});
