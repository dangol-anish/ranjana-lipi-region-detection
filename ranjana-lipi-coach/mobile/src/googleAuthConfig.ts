export const GOOGLE_AUTH_CONFIG = {
  expoClientId:
    "528858819516-bbq1ilob45o7ihfkr0s4o89svi7qagh0.apps.googleusercontent.com",
  iosClientId: "",
  // Required for Google sign-in on Android phones. Create an Android OAuth
  // client in Google Cloud/Firebase and paste it here.
  androidClientId:
    "528858819516-u962ubb63t956g39el3777qoj8ffucmc.apps.googleusercontent.com",
  webClientId:
    "528858819516-bbq1ilob45o7ihfkr0s4o89svi7qagh0.apps.googleusercontent.com",
};

export function isGoogleAuthConfigured(): boolean {
  return Object.values(GOOGLE_AUTH_CONFIG).some(
    (value) => value.trim().length > 0,
  );
}
