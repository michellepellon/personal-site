// Cookie Consent Management
(function() {
  'use strict';

  // Check if user has already consented
  function hasConsent() {
    return localStorage.getItem('cookieConsent') === 'accepted';
  }

  // Create cookie banner
  function createBanner() {
    const banner = document.createElement('div');
    banner.id = 'cookie-consent-banner';
    banner.setAttribute('role', 'region');
    banner.setAttribute('aria-label', 'Cookie consent');
    banner.setAttribute('aria-live', 'polite');
    
    banner.innerHTML = `
      <style>
        #cookie-consent-banner {
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          background: #1a1a1a;
          color: white;
          padding: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          z-index: 1000;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          font-size: 0.875rem;
          box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        #cookie-consent-banner p {
          margin: 0;
          flex: 1;
        }
        
        #cookie-consent-banner a {
          color: #66b3ff;
          text-decoration: underline;
        }
        
        #cookie-consent-banner a:hover {
          color: #99ccff;
        }
        
        .cookie-buttons {
          display: flex;
          gap: 0.5rem;
        }
        
        .cookie-button {
          padding: 0.5rem 1rem;
          border: none;
          cursor: pointer;
          font-size: 0.875rem;
          border-radius: 4px;
          transition: background-color 0.2s ease;
          font-family: inherit;
        }
        
        .cookie-button-accept {
          background: rgb(56, 113, 222);
          color: white;
        }
        
        .cookie-button-accept:hover {
          background: rgb(40, 90, 180);
        }
        
        .cookie-button-reject {
          background: transparent;
          color: white;
          border: 1px solid #666;
        }
        
        .cookie-button-reject:hover {
          background: rgba(255,255,255,0.1);
        }
        
        @media (max-width: 600px) {
          #cookie-consent-banner {
            flex-direction: column;
            text-align: center;
          }
          
          .cookie-buttons {
            width: 100%;
            justify-content: center;
          }
        }
      </style>
      <p>
        We use cookies to analyze site traffic and improve your experience. 
        <a href="/privacy.html">Learn more in our Privacy Policy</a>.
      </p>
      <div class="cookie-buttons">
        <button class="cookie-button cookie-button-reject" onclick="rejectCookies()">Reject</button>
        <button class="cookie-button cookie-button-accept" onclick="acceptCookies()">Accept</button>
      </div>
    `;
    
    document.body.appendChild(banner);
  }

  // Accept cookies
  window.acceptCookies = function() {
    localStorage.setItem('cookieConsent', 'accepted');
    localStorage.setItem('cookieConsentDate', new Date().toISOString());
    document.getElementById('cookie-consent-banner').remove();
    
    // Initialize Google Analytics
    if (typeof gtag !== 'undefined') {
      gtag('consent', 'update', {
        'analytics_storage': 'granted'
      });
    }
  };

  // Reject cookies
  window.rejectCookies = function() {
    localStorage.setItem('cookieConsent', 'rejected');
    localStorage.setItem('cookieConsentDate', new Date().toISOString());
    document.getElementById('cookie-consent-banner').remove();
    
    // Disable Google Analytics
    if (typeof gtag !== 'undefined') {
      gtag('consent', 'update', {
        'analytics_storage': 'denied'
      });
    }
  };

  // Initialize on page load
  document.addEventListener('DOMContentLoaded', function() {
    if (!hasConsent() && localStorage.getItem('cookieConsent') !== 'rejected') {
      // Set default consent state to denied
      if (typeof gtag !== 'undefined') {
        gtag('consent', 'default', {
          'analytics_storage': 'denied'
        });
      }
      createBanner();
    } else if (hasConsent() && typeof gtag !== 'undefined') {
      // User has already consented
      gtag('consent', 'update', {
        'analytics_storage': 'granted'
      });
    }
  });
})();