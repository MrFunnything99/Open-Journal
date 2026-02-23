import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import { Personaplex } from "./pages/Personaplex/Personaplex";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Personaplex />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <RouterProvider router={router}/>
);
